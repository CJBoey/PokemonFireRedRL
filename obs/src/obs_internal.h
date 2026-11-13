/* fireredsim observer: internal declarations (C, compiled with the game's flags).
 *
 * Static (file-local) game functions and variables cannot be named from C, so the
 * observer resolves them once from the executable's symbol table, file-qualified
 * ("file.c:name", obs_resolve.cpp). A symbol that cannot be resolved stays 0; every
 * predicate that needs it is then simply never true (a false negative, never a
 * false positive), and pn_obs_selfcheck() reports it.
 *
 * Function pointers stored in gMain/gTasks/controllers always equal the canonical
 * symbol (docs/OBS_SOURCES.md section 0.4). RAM statics sit at their retail
 * addresses (mirror layout) and are dereferenced through the resolved address.
 */
#ifndef PN_OBS_INTERNAL_H
#define PN_OBS_INTERNAL_H

#include <stdint.h>

/* X(id, "file.c:name") */
#define PN_OBS_SYMBOLS(X) \
    /* --- callbacks --- */ \
    X(CB2_Overworld, "overworld.c:CB2_Overworld") \
    X(CB2_OverworldBasic, "overworld.c:CB2_OverworldBasic") \
    X(CB1_Overworld, "overworld.c:CB1_Overworld") \
    X(CB1_UpdateLinkState, "overworld.c:CB1_UpdateLinkState") \
    X(CB2_LoadMap, "overworld.c:CB2_LoadMap") \
    X(CB2_DoChangeMap, "fldeff_flash.c:CB2_DoChangeMap") \
    X(CB2_ChangeMapMain, "fldeff_flash.c:CB2_ChangeMapMain") \
    X(CB2_LoadMap2, "overworld.c:CB2_LoadMap2") \
    X(CB2_ReturnToField, "overworld.c:CB2_ReturnToField") \
    X(CB2_ReturnToFieldLocal, "overworld.c:CB2_ReturnToFieldLocal") \
    X(BattleMainCB1, "battle_main.c:BattleMainCB1") \
    X(BattleMainCB2, "battle_main.c:BattleMainCB2") \
    X(CB2_TitleScreenRun, "title_screen.c:CB2_TitleScreenRun") \
    X(CB2_MainMenu, "main_menu.c:CB2_MainMenu") \
    X(CB2_NewGameScene, "oak_speech.c:CB2_NewGameScene") \
    X(CB2_NamingScreen, "naming_screen.c:CB2_NamingScreen") \
    X(CB2_BagMenuRun, "item_menu.c:CB2_BagMenuRun") \
    X(CB2_UpdatePartyMenu, "party_menu.c:CB2_UpdatePartyMenu") \
    X(CB2_RunPokemonSummaryScreen, "pokemon_summary_screen.c:CB2_RunPokemonSummaryScreen") \
    X(CB2_TrainerCard, "trainer_card.c:CB2_TrainerCard") \
    X(CB2_InitOptionMenu, "option_menu.c:CB2_InitOptionMenu") \
    X(CB2_BuyMenu, "shop.c:CB2_BuyMenu") \
    X(CB2_EvolutionSceneUpdate, "evolution_scene.c:CB2_EvolutionSceneUpdate") \
    X(CB2_TradeEvolutionSceneUpdate, "evolution_scene.c:CB2_TradeEvolutionSceneUpdate") \
    X(CB2_EggHatch_1, "daycare.c:CB2_EggHatch_1") \
    X(CB2_InGameTrade, "trade_scene.c:CB2_InGameTrade") \
    X(CB2_PSA, "pokemon_special_anim.c:CB2_PSA") \
    X(CB2_RegionMap, "region_map.c:CB2_RegionMap") \
    X(CB2_OpenRegionMap, "region_map.c:CB2_OpenRegionMap") \
    /* --- overworld / script tasks and natives --- */ \
    X(Task_BattleStart, "battle_setup.c:Task_BattleStart") \
    X(Task_Teleport2Warp, "field_fadetransition.c:Task_Teleport2Warp") \
    X(Task_TeleportWarp, "field_fadetransition.c:Task_TeleportWarp") \
    X(Task_DoorWarp, "field_fadetransition.c:Task_DoorWarp") \
    X(Task_StairWarp, "field_fadetransition.c:Task_StairWarp") \
    X(Task_ExitDoor, "field_fadetransition.c:Task_ExitDoor") \
    X(Task_ExitNonAnimDoor, "field_fadetransition.c:Task_ExitNonAnimDoor") \
    X(Task_ExitNonDoor, "field_fadetransition.c:Task_ExitNonDoor") \
    X(Task_TeleportWarpIn, "field_fadetransition.c:Task_TeleportWarpIn") \
    X(Task_ExitStairs, "field_fadetransition.c:Task_ExitStairs") \
    X(Task_DrawFieldMessageBox, "field_message_box.c:Task_DrawFieldMessageBox") \
    X(WaitForAorBPress, "scrcmd.c:WaitForAorBPress") \
    X(Task_YesNoMenu_HandleInput, "script_menu.c:Task_YesNoMenu_HandleInput") \
    X(Task_MultichoiceMenu_HandleInput, "script_menu.c:Task_MultichoiceMenu_HandleInput") \
    X(Hask_MultichoiceGridMenu_HandleInput, "script_menu.c:Hask_MultichoiceGridMenu_HandleInput") \
    X(Task_ListMenuHandleInput, "field_specials.c:Task_ListMenuHandleInput") \
    X(Task_StartMenuHandleInput, "start_menu.c:Task_StartMenuHandleInput") \
    X(task50_startmenu, "start_menu.c:task50_startmenu") \
    X(Task_WaitFadeAndCreateStartMenuTask, "field_fadetransition.c:Task_WaitFadeAndCreateStartMenuTask") \
    X(task50_save_game, "start_menu.c:task50_save_game") \
    X(StartCB_HandleInput, "start_menu.c:StartCB_HandleInput") \
    X(StartCB_Save1, "start_menu.c:StartCB_Save1") \
    X(StartCB_Save2, "start_menu.c:StartCB_Save2") \
    X(SaveDialogCB_AskSaveHandleInput, "start_menu.c:SaveDialogCB_AskSaveHandleInput") \
    X(SaveDialogCB_AskOverwriteOrReplacePreviousFileHandleInput, "start_menu.c:SaveDialogCB_AskOverwriteOrReplacePreviousFileHandleInput") \
    X(Task_ShopMenu, "shop.c:Task_ShopMenu") \
    X(Task_ReturnToShopMenu, "shop.c:Task_ReturnToShopMenu") \
    X(Task_BuyMenu, "shop.c:Task_BuyMenu") \
    X(Task_BuyHowManyDialogueHandleInput, "shop.c:Task_BuyHowManyDialogueHandleInput") \
    X(Task_ReturnToItemListAfterItemPurchase, "shop.c:Task_ReturnToItemListAfterItemPurchase") \
    X(Task_PCMainMenu, "pokemon_storage_system_menu.c:Task_PCMainMenu") \
    X(Task_TopMenuHandleInput_pc, "player_pc.c:Task_TopMenuHandleInput") \
    X(Task_TopMenu_ItemStorageSubmenu_HandleInput, "player_pc.c:Task_TopMenu_ItemStorageSubmenu_HandleInput") \
    X(Task_MailSubmenuHandleInput, "player_pc.c:Task_MailSubmenuHandleInput") \
    X(Task_MailboxPcHandleInput, "player_pc.c:Task_MailboxPcHandleInput") \
    X(Task_MoveToBagYesNoMenuHandleInput, "player_pc.c:Task_MoveToBagYesNoMenuHandleInput") \
    X(ItemPc_MainCB, "item_pc.c:ItemPc_MainCB") \
    X(Task_ItemPcMain, "item_pc.c:Task_ItemPcMain") \
    X(Task_ItemPcSubmenuRun, "item_pc.c:Task_ItemPcSubmenuRun") \
    X(Task_ItemPcMoveItemModeRun, "item_pc.c:Task_ItemPcMoveItemModeRun") \
    X(Task_ItemPcHandleWithdrawMultiple, "item_pc.c:Task_ItemPcHandleWithdrawMultiple") \
    X(Task_ItemPcWaitButtonAndFinishWithdrawMultiple, "item_pc.c:Task_ItemPcWaitButtonAndFinishWithdrawMultiple") \
    X(Task_ItemPcWaitButtonWithdrawMultipleFailed, "item_pc.c:Task_ItemPcWaitButtonWithdrawMultipleFailed") \
    X(Task_PCScreenEffect_TurnOn, "pc_screen_effect.c:Task_PCScreenEffect_TurnOn") \
    X(Task_ContinueTaskAfterMessagePrints, "menu_helpers.c:Task_ContinueTaskAfterMessagePrints") \
    X(Task_CallYesOrNoCallback, "menu_helpers.c:Task_CallYesOrNoCallback") \
    X(Task_QuestLogPlayback_OpenStartMenu, "field_control_avatar.c:Task_QuestLogPlayback_OpenStartMenu") \
    X(Task_WaitAtEndOfQuestLog, "quest_log.c:Task_WaitAtEndOfQuestLog") \
    /* --- title, main menu, Oak speech, naming --- */ \
    X(Task_TitleScreenMain, "title_screen.c:Task_TitleScreenMain") \
    X(Task_HandleMenuInput_main, "main_menu.c:Task_HandleMenuInput") \
    X(Task_ControlsGuide_HandleInput, "oak_speech.c:Task_ControlsGuide_HandleInput") \
    X(Task_PikachuIntro_HandleInput, "oak_speech.c:Task_PikachuIntro_HandleInput") \
    X(Task_OakSpeech_HandleGenderInput, "oak_speech.c:Task_OakSpeech_HandleGenderInput") \
    X(Task_OakSpeech_HandleRivalNameInput, "oak_speech.c:Task_OakSpeech_HandleRivalNameInput") \
    X(Task_OakSpeech_HandleConfirmNameInput, "oak_speech.c:Task_OakSpeech_HandleConfirmNameInput") \
    X(Task_HandleInput_naming, "naming_screen.c:Task_HandleInput") \
    /* --- bag --- */ \
    X(Task_BagMenu_HandleInput, "item_menu.c:Task_BagMenu_HandleInput") \
    X(Task_AnimateWin0v, "item_menu.c:Task_AnimateWin0v") \
    X(Task_MoveItemInPocket_HandleInput, "item_menu.c:Task_MoveItemInPocket_HandleInput") \
    X(Task_FieldItemContextMenuHandleInput, "item_menu.c:Task_FieldItemContextMenuHandleInput") \
    X(Task_SelectQuantityToToss, "item_menu.c:Task_SelectQuantityToToss") \
    X(Task_SelectQuantityToSell, "item_menu.c:Task_SelectQuantityToSell") \
    X(Task_SelectQuantityToDeposit, "item_menu.c:Task_SelectQuantityToDeposit") \
    X(Task_WaitAB_RedrawAndReturnToBag, "item_menu.c:Task_WaitAB_RedrawAndReturnToBag") \
    X(Task_WaitPressAB_AfterSell, "item_menu.c:Task_WaitPressAB_AfterSell") \
    X(Task_WaitAButtonAndCloseContextMenu, "item_menu.c:Task_WaitAButtonAndCloseContextMenu") \
    /* --- party menu, summary --- */ \
    X(Task_HandleChooseMonInput, "party_menu.c:Task_HandleChooseMonInput") \
    X(Task_HandleSelectionMenuInput, "party_menu.c:Task_HandleSelectionMenuInput") \
    X(Task_HandleCancelChooseMonYesNoInput, "party_menu.c:Task_HandleCancelChooseMonYesNoInput") \
    X(Task_HandleCancelParticipationYesNoInput, "party_menu.c:Task_HandleCancelParticipationYesNoInput") \
    X(Task_HandleSwitchItemsYesNoInput, "party_menu.c:Task_HandleSwitchItemsYesNoInput") \
    X(Task_HandleSendMailToPCYesNoInput, "party_menu.c:Task_HandleSendMailToPCYesNoInput") \
    X(Task_HandleLoseMailMessageYesNoInput, "party_menu.c:Task_HandleLoseMailMessageYesNoInput") \
    X(Task_HandleFieldMoveExitAreaYesNoInput, "party_menu.c:Task_HandleFieldMoveExitAreaYesNoInput") \
    X(Task_HandleReplaceMoveYesNoInput, "party_menu.c:Task_HandleReplaceMoveYesNoInput") \
    X(Task_HandleStopLearningMoveYesNoInput, "party_menu.c:Task_HandleStopLearningMoveYesNoInput") \
    X(Task_HandleSwitchItemsFromBagYesNoInput, "party_menu.c:Task_HandleSwitchItemsFromBagYesNoInput") \
    X(Task_HandleRestoreWhichMoveInput, "party_menu.c:Task_HandleRestoreWhichMoveInput") \
    X(Task_CancelAfterAorBPress, "party_menu.c:Task_CancelAfterAorBPress") \
    X(Task_DisplayLevelUpStatsPg2, "party_menu.c:Task_DisplayLevelUpStatsPg2") \
    X(Task_PrintAndWaitForText, "party_menu.c:Task_PrintAndWaitForText") \
    X(Task_FirstBattleEnterParty_RunPrinterMsg1, "party_menu.c:Task_FirstBattleEnterParty_RunPrinterMsg1") \
    X(Task_FirstBattleEnterParty_RunPrinterMsg2, "party_menu.c:Task_FirstBattleEnterParty_RunPrinterMsg2") \
    X(Task_InputHandler_Info, "pokemon_summary_screen.c:Task_InputHandler_Info") \
    X(Task_PokeSum_FlipPages, "pokemon_summary_screen.c:Task_PokeSum_FlipPages") \
    X(Task_PokeSum_SwitchDisplayedPokemon, "pokemon_summary_screen.c:Task_PokeSum_SwitchDisplayedPokemon") \
    X(Task_HandleInput_SelectMove, "pokemon_summary_screen.c:Task_HandleInput_SelectMove") \
    X(Task_InputHandler_SelectOrForgetMove, "pokemon_summary_screen.c:Task_InputHandler_SelectOrForgetMove") \
    /* --- other full-screen menus --- */ \
    X(Task_TrainerCard, "trainer_card.c:Task_TrainerCard") \
    X(Task_OptionMenu, "option_menu.c:Task_OptionMenu") \
    X(Task_EvolutionScene, "evolution_scene.c:Task_EvolutionScene") \
    /* --- battle --- */ \
    X(BeginBattleIntroDummy, "battle_main.c:BeginBattleIntroDummy") \
    X(BeginBattleIntro, "battle_main.c:BeginBattleIntro") \
    X(BattleIntroGetMonsData, "battle_main.c:BattleIntroGetMonsData") \
    X(BattleIntroPrepareBackgroundSlide, "battle_main.c:BattleIntroPrepareBackgroundSlide") \
    X(BattleIntroDrawTrainersOrMonsSprites, "battle_main.c:BattleIntroDrawTrainersOrMonsSprites") \
    X(RunBattleScriptCommands, "battle_main.c:RunBattleScriptCommands") \
    X(RunBattleScriptCommands_PopCallbacksStack, "battle_main.c:RunBattleScriptCommands_PopCallbacksStack") \
    X(HandleEndTurn_FinishBattle, "battle_main.c:HandleEndTurn_FinishBattle") \
    X(RunTurnActionsFunctions, "battle_main.c:RunTurnActionsFunctions") \
    X(FreeResetData_ReturnToOvOrDoEvolutions, "battle_main.c:FreeResetData_ReturnToOvOrDoEvolutions") \
    X(TryEvolvePokemon, "battle_main.c:TryEvolvePokemon") \
    X(WaitForEvoSceneToFinish, "battle_main.c:WaitForEvoSceneToFinish") \
    X(ReturnFromBattleToOverworld, "battle_main.c:ReturnFromBattleToOverworld") \
    X(HandleInputChooseAction_player, "battle_controller_player.c:HandleInputChooseAction") \
    X(HandleInputChooseAction_safari, "battle_controller_safari.c:HandleInputChooseAction") \
    X(HandleInputChooseAction_oak, "battle_controller_oak_old_man.c:HandleInputChooseAction") \
    X(HandleInputChooseMove, "battle_controller_player.c:HandleInputChooseMove") \
    X(OakOldManHandleInputChooseMove, "battle_controller_oak_old_man.c:OakOldManHandleInputChooseMove") \
    X(HandleInputChooseTarget, "battle_controller_player.c:HandleInputChooseTarget") \
    X(HandleMoveSwitching, "battle_controller_player.c:HandleMoveSwitching") \
    X(WaitForMonSelection_player, "battle_controller_player.c:WaitForMonSelection") \
    X(WaitForMonSelection_oak, "battle_controller_oak_old_man.c:WaitForMonSelection") \
    X(CompleteWhenChoseItem_player, "battle_controller_player.c:CompleteWhenChoseItem") \
    X(CompleteWhenChoseItem_oak, "battle_controller_oak_old_man.c:CompleteWhenChoseItem") \
    X(Cmd_yesnobox, "battle_script_commands.c:Cmd_yesnobox") \
    X(Cmd_yesnoboxlearnmove, "battle_script_commands.c:Cmd_yesnoboxlearnmove") \
    X(Cmd_yesnoboxstoplearningmove, "battle_script_commands.c:Cmd_yesnoboxstoplearningmove") \
    X(Cmd_trygivecaughtmonnick, "battle_script_commands.c:Cmd_trygivecaughtmonnick") \
    X(Cmd_drawlvlupbox, "battle_script_commands.c:Cmd_drawlvlupbox") \
    X(Task_DexScreen_RegisterMonToPokedex, "pokedex_screen.c:Task_DexScreen_RegisterMonToPokedex") \
    /* --- v2: Pokedex --- */ \
    X(CB2_PokedexScreen, "pokedex_screen.c:CB2_PokedexScreen") \
    X(Task_PokedexScreen, "pokedex_screen.c:Task_PokedexScreen") \
    X(Task_DexScreen_NumericalOrder, "pokedex_screen.c:Task_DexScreen_NumericalOrder") \
    X(Task_DexScreen_CharacteristicOrder, "pokedex_screen.c:Task_DexScreen_CharacteristicOrder") \
    X(Task_DexScreen_CategorySubmenu, "pokedex_screen.c:Task_DexScreen_CategorySubmenu") \
    X(Task_DexScreen_ShowMonPage, "pokedex_screen.c:Task_DexScreen_ShowMonPage") \
    /* --- v2: Pokemon storage (PC box) --- */ \
    X(CB2_PokeStorage, "pokemon_storage_system_tasks.c:CB2_PokeStorage") \
    X(Task_PokeStorageMain, "pokemon_storage_system_tasks.c:Task_PokeStorageMain") \
    X(Task_OnSelectedMon, "pokemon_storage_system_tasks.c:Task_OnSelectedMon") \
    X(Task_WithdrawMon, "pokemon_storage_system_tasks.c:Task_WithdrawMon") \
    X(Task_DepositMenu, "pokemon_storage_system_tasks.c:Task_DepositMenu") \
    X(Task_ReleaseMon, "pokemon_storage_system_tasks.c:Task_ReleaseMon") \
    X(Task_ShowMarkMenu, "pokemon_storage_system_tasks.c:Task_ShowMarkMenu") \
    X(Task_GiveMovingItemToMon, "pokemon_storage_system_tasks.c:Task_GiveMovingItemToMon") \
    X(Task_ItemToBag, "pokemon_storage_system_tasks.c:Task_ItemToBag") \
    X(Task_SwitchSelectedItem, "pokemon_storage_system_tasks.c:Task_SwitchSelectedItem") \
    X(Task_ShowItemInfo, "pokemon_storage_system_tasks.c:Task_ShowItemInfo") \
    X(Task_CloseBoxWhileHoldingItem, "pokemon_storage_system_tasks.c:Task_CloseBoxWhileHoldingItem") \
    X(Task_PrintCantStoreMail, "pokemon_storage_system_tasks.c:Task_PrintCantStoreMail") \
    X(Task_HandleBoxOptions, "pokemon_storage_system_tasks.c:Task_HandleBoxOptions") \
    X(Task_HandleWallpapers, "pokemon_storage_system_tasks.c:Task_HandleWallpapers") \
    X(Task_JumpBox, "pokemon_storage_system_tasks.c:Task_JumpBox") \
    X(Task_OnCloseBoxPressed, "pokemon_storage_system_tasks.c:Task_OnCloseBoxPressed") \
    X(Task_OnBPressed, "pokemon_storage_system_tasks.c:Task_OnBPressed") \
    /* --- v2: region map / Fly map --- */ \
    X(Task_RegionMap, "region_map.c:Task_RegionMap") \
    X(Task_FlyMap, "region_map.c:Task_FlyMap") \
    X(HandleRegionMapInput, "region_map.c:HandleRegionMapInput") \
    X(CB2_OpenFlyMap, "region_map.c:CB2_OpenFlyMap") \
    /* --- v2: TM case, berry pouch --- */ \
    X(CB2_Idle_tm, "tm_case.c:CB2_Idle") \
    X(Task_HandleListInput_tm, "tm_case.c:Task_HandleListInput") \
    X(Task_ContextMenu_HandleInput_tm, "tm_case.c:Task_ContextMenu_HandleInput") \
    X(Task_QuantitySelect_HandleInput_tm, "tm_case.c:Task_QuantitySelect_HandleInput") \
    X(Task_AfterSale_ReturnToList_tm, "tm_case.c:Task_AfterSale_ReturnToList") \
    X(Task_WaitButtonAfterErrorPrint_tm, "tm_case.c:Task_WaitButtonAfterErrorPrint") \
    X(CB2_BerryPouchIdle, "berry_pouch.c:CB2_BerryPouchIdle") \
    X(Task_BerryPouchMain, "berry_pouch.c:Task_BerryPouchMain") \
    X(Task_NormalContextMenu_HandleInput_brp, "berry_pouch.c:Task_NormalContextMenu_HandleInput") \
    X(Task_Toss_SelectMultiple_brp, "berry_pouch.c:Task_Toss_SelectMultiple") \
    X(Task_Sell_SelectMultiple_brp, "berry_pouch.c:Task_Sell_SelectMultiple") \
    X(Task_WaitButtonThenTossBerries_brp, "berry_pouch.c:Task_WaitButtonThenTossBerries") \
    X(Task_SellBerries_WaitButton_brp, "berry_pouch.c:Task_SellBerries_WaitButton") \
    X(Task_WaitButtonBeforeDialogueWindowDestruction_brp, "berry_pouch.c:Task_WaitButtonBeforeDialogueWindowDestruction") \
    /* --- v2: move relearner, main menu messages --- */ \
    X(CB2_MoveRelearner, "learn_move.c:CB2_MoveRelearner") \
    X(Task_SaveErrorStatus_RunPrinterThenWaitButton, "main_menu.c:Task_SaveErrorStatus_RunPrinterThenWaitButton") \
    X(Task_MysteryGiftError, "main_menu.c:Task_MysteryGiftError") \
    X(CB2_HofIdle, "hall_of_fame.c:CB2_HofIdle") \
    X(Task_Hof_ExitOnKeyPressed, "hall_of_fame.c:Task_Hof_ExitOnKeyPressed") \
    X(Task_HofPC_HandleInput, "hall_of_fame.c:Task_HofPC_HandleInput") \
    X(Task_HofPC_ExitOnButtonPress, "hall_of_fame.c:Task_HofPC_ExitOnButtonPress") \
    /* --- v3.1 --- */ \
    X(Task_RushInjuredPokemonToCenter, "field_screen_effect.c:Task_RushInjuredPokemonToCenter") \
    /* --- v3.2: battle records, Teachy TV, Pokedude demos, item-use animation --- */ \
    X(MainCB2_records, "battle_records.c:MainCB2") \
    X(Task_WaitButton_records, "battle_records.c:Task_WaitButton") \
    X(TeachyTvCallback, "teachy_tv.c:TeachyTvCallback") \
    X(TeachyTvOptionListController, "teachy_tv.c:TeachyTvOptionListController") \
    X(TeachyTvRenderMsgAndSwitchClusterFuncs, "teachy_tv.c:TeachyTvRenderMsgAndSwitchClusterFuncs") \
    X(TTVcmd_IdleIfTextPrinterIsActive, "teachy_tv.c:TTVcmd_IdleIfTextPrinterIsActive") \
    X(TTVcmd_IdleIfTextPrinterIsActive2, "teachy_tv.c:TTVcmd_IdleIfTextPrinterIsActive2") \
    X(TTVcmd_EraseTextWindowIfKeyPressed, "teachy_tv.c:TTVcmd_EraseTextWindowIfKeyPressed") \
    X(sBattleScript_ttv, "teachy_tv.c:sBattleScript") \
    X(sStatusScript_ttv, "teachy_tv.c:sStatusScript") \
    X(sMatchupsScript_ttv, "teachy_tv.c:sMatchupsScript") \
    X(sCatchingScript_ttv, "teachy_tv.c:sCatchingScript") \
    X(sTMsScript_ttv, "teachy_tv.c:sTMsScript") \
    X(sRegisterKeyItemScript_ttv, "teachy_tv.c:sRegisterKeyItemScript") \
    X(CB2_BagMenuFromStartMenu, "item_menu.c:CB2_BagMenuFromStartMenu") \
    X(PokedudeAction_PrintVoiceoverMessage, "battle_controller_pokedude.c:PokedudeAction_PrintVoiceoverMessage") \
    X(PokedudeAction_PrintMessageWithHealthboxPals, "battle_controller_pokedude.c:PokedudeAction_PrintMessageWithHealthboxPals") \
    X(Pokedude_SetBattleEndCallbacks, "battle_controller_pokedude.c:Pokedude_SetBattleEndCallbacks") \
    X(OpenPartyMenuToChooseMon_pokedude, "battle_controller_pokedude.c:OpenPartyMenuToChooseMon") \
    X(OpenBagAndChooseItem_pokedude, "battle_controller_pokedude.c:OpenBagAndChooseItem") \
    X(Task_Bag_TeachyTvRegister, "item_menu.c:Task_Bag_TeachyTvRegister") \
    X(Task_Bag_TeachyTvCatching, "item_menu.c:Task_Bag_TeachyTvCatching") \
    X(Task_Bag_TeachyTvStatus, "item_menu.c:Task_Bag_TeachyTvStatus") \
    X(Task_Bag_TeachyTvTMs, "item_menu.c:Task_Bag_TeachyTvTMs") \
    X(Task_PartyMenu_PokedudeStep, "party_menu.c:Task_PartyMenu_PokedudeStep") \
    X(Task_PartyMenuFromBag_PokedudeStep, "party_menu.c:Task_PartyMenuFromBag_PokedudeStep") \
    X(Task_Pokedude_Run_tm, "tm_case.c:Task_Pokedude_Run") \
    X(Task_UseItem_Normal_psa, "pokemon_special_anim.c:Task_UseItem_Normal") \
    X(Task_EvoStone_CantEvolve_psa, "pokemon_special_anim.c:Task_EvoStone_CantEvolve") \
    /* --- RAM statics --- */ \
    X(sTextPrinters, "text_printer.c:sTextPrinters") \
    X(sMenu, "menu.c:sMenu") \
    X(sYesNoWindowId, "menu.c:sYesNoWindowId") \
    X(sGlobalScriptContextStatus, "script.c:sGlobalScriptContextStatus") \
    X(sGlobalScriptContext, "script.c:sGlobalScriptContext") \
    X(sStartMenuCallback, "start_menu.c:sStartMenuCallback") \
    X(sStartMenuCursorPos, "start_menu.c:sStartMenuCursorPos") \
    X(sNumStartMenuItems, "start_menu.c:sNumStartMenuItems") \
    X(sStartMenuOrder, "start_menu.c:sStartMenuOrder") \
    X(sSaveDialogCB, "start_menu.c:sSaveDialogCB") \
    X(sVideoState, "help_system_util.c:sVideoState") \
    X(sDelayTimer_help, "help_system_util.c:sDelayTimer") \
    X(sSaveFailedScreenState, "save_failed_screen.c:sSaveFailedScreenState") \
    X(sIsInSaveFailedScreen, "save_failed_screen.c:sIsInSaveFailedScreen") \
    X(sSpecialFlags, "event_data.c:sSpecialFlags") \
    X(sDelay_scriptmenu, "script_menu.c:sDelay") \
    X(sOakSpeechResources, "oak_speech.c:sOakSpeechResources") \
    X(sNamingScreen, "naming_screen.c:sNamingScreen") \
    X(sTrainerCardDataPtr, "trainer_card.c:sTrainerCardDataPtr") \
    X(sOptionMenuPtr, "option_menu.c:sOptionMenuPtr") \
    X(sMonSummaryScreen, "pokemon_summary_screen.c:sMonSummaryScreen") \
    X(sPartyMenuInternal, "party_menu.c:sPartyMenuInternal") \
    X(sBagMenuDisplay, "item_menu.c:sBagMenuDisplay") \
    X(sPlayTimeCounterState, "play_time.c:sPlayTimeCounterState") \
    X(sPlaybackControl, "quest_log.c:sPlaybackControl") \
    X(sFanfareCounter, "sound.c:sFanfareCounter") \
    X(sWildEncounterData, "wild_encounter.c:sWildEncounterData") \
    X(sMessageWindowId, "menu_helpers.c:sMessageWindowId") \
    X(sMessageNextTask, "menu_helpers.c:sMessageNextTask") \
    X(sForcedMovementFuncs, "field_player_avatar.c:sForcedMovementFuncs") \
    /* --- v2 RAM statics --- */ \
    X(sPokedexScreenData, "pokedex_screen.c:sPokedexScreenData") \
    X(sCursorArea_pss, "pokemon_storage_system_data.c:sCursorArea") \
    X(sCursorPosition_pss, "pokemon_storage_system_data.c:sCursorPosition") \
    X(sRegionMap, "region_map.c:sRegionMap") \
    X(sMapCursor, "region_map.c:sMapCursor") \
    X(sFlyMap, "region_map.c:sFlyMap") \
    X(sTMCaseStaticResources, "tm_case.c:sTMCaseStaticResources") \
    X(sStaticCnt_brp, "berry_pouch.c:sStaticCnt") \
    X(sMoveRelearner, "learn_move.c:sMoveRelearner") \
    /* --- v3.2 RAM statics --- */ \
    X(sStaticResources_ttv, "teachy_tv.c:sStaticResources") \
    /* --- v4: menu actions, item PC, PC boxes, mart --- */ \
    X(sContextMenuItemsPtr, "item_menu.c:sContextMenuItemsPtr") \
    X(sContextMenuNumItems, "item_menu.c:sContextMenuNumItems") \
    X(sStateDataPtr_itempc, "item_pc.c:sStateDataPtr") \
    X(sListMenuState_itempc, "item_pc.c:sListMenuState") \
    X(sIsMonBeingMoved_pss, "pokemon_storage_system_data.c:sIsMonBeingMoved") \
    X(sShopData, "shop.c:sShopData") \
    /* --- v4: TM case, berry pouch --- */ \
    X(sTMCaseDynamicResources, "tm_case.c:sTMCaseDynamicResources") \
    X(sResources_brp, "berry_pouch.c:sResources") \
    X(sContextMenuOptions_brp, "berry_pouch.c:sContextMenuOptions") \
    X(sContextMenuNumOptions_brp, "berry_pouch.c:sContextMenuNumOptions") \
    /* --- v7: learn-move prompts --- */ \
    X(sLastViewedMonIndex_pss, "pokemon_summary_screen.c:sLastViewedMonIndex") \
    /* --- v8: the Summary moves page cursor --- */ \
    X(sMoveSelectionCursorPos_pss, "pokemon_summary_screen.c:sMoveSelectionCursorPos")

enum PnObsSymId {
#define PN_OBS_SYM_ENUM(id, name) PNSYM_##id,
    PN_OBS_SYMBOLS(PN_OBS_SYM_ENUM)
#undef PN_OBS_SYM_ENUM
    PNSYM_COUNT
};

extern uint32_t gPnObsSym[PNSYM_COUNT];
#define SYM(id) (gPnObsSym[PNSYM_##id])
#define SYMPTR(type, id) ((type)(uintptr_t)gPnObsSym[PNSYM_##id])

/* obs_resolve.cpp */
int pn_obs_resolve(const char *const *names, uint32_t *out, int n, int verbose);
void pn_obs_host_info(uint64_t *frame, uint8_t *soft_reset);

/* observe.c: resolves symbols and the screen table once */
void pn_obs_init(void);

/* wait.c */
typedef struct PnWait {
    uint8_t waiting;       /* waiting_for_input */
    uint8_t kind;          /* enum PnInputKind */
    uint8_t ctx;           /* enum PnWaitCtx */
    uint8_t flags;         /* PN_INPUT_FLAG_* */
    uint16_t keys;         /* input_keys */
    uint8_t screen;        /* enum PnScreen */
    uint8_t menu_open;     /* enum PnMenuOpen */
    uint8_t menu_cursor;
    uint8_t menu_count;
    uint8_t menu_id;
    uint8_t yesno_cursor;  /* battle/evolution YES/NO */
    uint8_t field_state;   /* enum PnFieldState */
    uint8_t gate_ok;       /* global gate G of section 2.9 */
    uint8_t help_open;
    uint8_t save_failed;
    uint8_t link;
    uint8_t ql_playback;
    uint8_t battle_input_battler; /* 255 none */
    int8_t whiteout_win;   /* v3.1: window of the whiteout message (WHITEOUT_MSG, in_dialog), -1 none */
    /* v4 (docs/OBSERVATION.md "Changes in v4") */
    uint8_t menu_action_count;              /* entries of menu_actions (0: no action list) */
    uint8_t menu_actions[PN_OBS_MENU_ACTIONS]; /* enum PnMenuAction, top to bottom */
    uint8_t box_choose_cursor;              /* box chooser cursor, 255 none */
    uint16_t cursor_item;                   /* item under a list cursor */
    uint16_t context_item;                  /* item a context menu / quantity / YES-NO / message acts on */
    uint16_t quantity;                      /* QUANTITY waits */
    uint16_t quantity_max;
    /* v7 (docs/OBSERVATION.md "Changes in v7"): the move offered by a learn-move YES/NO or the Summary forget
     * screen and the party slot of the mon concerned (both set or neither: 0 / 255) */
    uint16_t offered_move;
    uint8_t offered_slot;
    uint8_t summary_cursor;  /* v8: 1 + sMoveSelectionCursorPos on the Summary moves page waits, 0 elsewhere */
    uint8_t party_cursor;    /* v10: 1 + the slot a party menu wait acts on (0..7), 0 elsewhere */
} PnWait;

void pn_obs_wait(PnWait *w, int soft_reset);
int pn_obs_screen_table_lookup(uint32_t cb2);   /* PnScreen or -1 */
int pn_obs_task_find(uint32_t fn);              /* task id or -1 (active task with that func) */
int pn_obs_ingame(void);
int pn_obs_raw_flag(uint16_t id);
int pn_obs_quest_log_playback(void);

/* field / battle gates shared with observe.c */
int pn_obs_field_gate(void);
int pn_obs_battle_gate(void);
int pn_obs_battle_mons_gate(void);

#endif
